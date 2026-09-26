// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package main

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"strings"
	"time"

	"github.com/GetStream/gophonic/chat"
	"github.com/GetStream/gophonic/mcp"
	"github.com/thesyncim/vibejson"
)

// Gopher's tools: what the model can do besides talking. Each is a Go
// function whose arguments struct tells the model how to call it; add one
// to give Gopher a new ability, or name an MCP server with -mcp.
func tools() []chat.Tool {
	return []chat.Tool{
		chat.Func("now", "The current date and time: here, in "+localZone()+", or in another time zone.", now),
		chat.Func("search", "Look something up in Wikipedia: people, places, events, facts, "+
			"and anything you may not know or that may have changed.", search),
	}
}

// localZone names the local time zone as IANA does, such as Europe/Lisbon.
func localZone() string {
	if tz := os.Getenv("TZ"); tz != "" {
		return tz
	}
	if link, err := os.Readlink("/etc/localtime"); err == nil {
		if _, name, ok := strings.Cut(link, "zoneinfo/"); ok {
			return name
		}
	}
	return time.Local.String()
}

// now tells the date and time.
func now(_ context.Context, args struct {
	Timezone string `json:"timezone,omitempty" desc:"an IANA time zone such as Asia/Tokyo, only when asked about somewhere else"`
}) (string, error) {
	loc := time.Local
	if args.Timezone != "" {
		var err error
		if loc, err = time.LoadLocation(args.Timezone); err != nil {
			return "", fmt.Errorf("unknown time zone %q", args.Timezone)
		}
	}
	t := time.Now().In(loc)
	zone, _ := t.Zone()
	name := loc.String()
	if loc == time.Local {
		name = localZone()
	}
	return t.Format("Monday, January 2, 2006, 3:04 PM") + " " + zone + " (" + name + ")", nil
}

// search looks a query up in Wikipedia: the best match's summary, and the
// titles of the next ones.
func search(ctx context.Context, args struct {
	Query    string `json:"query" desc:"what to look up"`
	Language string `json:"language,omitempty" desc:"the Wikipedia to search, as a language code such as en or pt; en if not given"`
}) (string, error) {
	lang := strings.ToLower(strings.TrimSpace(args.Language))
	if len(lang) < 2 || len(lang) > 3 {
		lang = "en"
	}
	// Full-text search ranks the pages, weighing how linked and read each
	// is, so "Eiffel Tower height" finds the tower before its replicas; the
	// best one's introduction, as plain text, is the answer's material.
	site := "https://" + lang + ".wikipedia.org/w/api.php?action=query&format=json&formatversion=2"
	var found struct {
		Query struct {
			Search []struct {
				Title string `json:"title"`
			} `json:"search"`
		} `json:"query"`
	}
	if err := getJSON(ctx, site+"&list=search&srlimit=3&srprop=&srqiprofile=wsum_inclinks_pv&srsearch="+url.QueryEscape(args.Query), &found); err != nil {
		return "", err
	}
	hits := found.Query.Search
	if len(hits) == 0 {
		return "Wikipedia has nothing on " + args.Query + ".", nil
	}
	var page struct {
		Query struct {
			Pages []struct {
				Title   string `json:"title"`
				Extract string `json:"extract"`
			} `json:"pages"`
		} `json:"query"`
	}
	if err := getJSON(ctx, site+"&prop=extracts&exintro=1&explaintext=1&redirects=1&titles="+url.QueryEscape(hits[0].Title), &page); err != nil {
		return "", err
	}
	if len(page.Query.Pages) == 0 {
		return "Wikipedia has nothing on " + args.Query + ".", nil
	}
	extract := page.Query.Pages[0].Extract
	if len(extract) > 1500 {
		extract = extract[:strings.LastIndexAny(extract[:1500], ".!?")+1]
	}
	var b strings.Builder
	fmt.Fprintf(&b, "Wikipedia, %s: %s", page.Query.Pages[0].Title, extract)
	for i, h := range hits[1:] {
		if i == 0 {
			b.WriteString(" Other articles:")
		}
		fmt.Fprintf(&b, " %s;", h.Title)
	}
	return b.String(), nil
}

// getJSON decodes the JSON at u into v.
func getJSON[T any](ctx context.Context, u string, v *T) error {
	ctx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, u, nil)
	if err != nil {
		return err
	}
	req.Header.Set("User-Agent", "gophonic-gopher/1.0 (https://github.com/GetStream/gophonic)")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	data, err := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if err != nil {
		return err
	}
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("%s: %s", u, resp.Status)
	}
	return vibejson.Unmarshal(data, v)
}

// connect starts each MCP server, a command line split at spaces, and adds
// its tools to tools. A tool whose name is taken is an error, not a shadow.
func connect(ctx context.Context, commands []string, tools []chat.Tool) ([]chat.Tool, []*mcp.Client, error) {
	var servers []*mcp.Client
	fail := func(err error) ([]chat.Tool, []*mcp.Client, error) {
		for _, s := range servers {
			s.Close()
		}
		return nil, nil, err
	}
	names := map[string]bool{}
	for _, t := range tools {
		names[t.Spec().Name] = true
	}
	for _, command := range commands {
		args := strings.Fields(command)
		cmd := exec.Command(args[0], args[1:]...)
		cmd.Stderr = os.Stderr
		server, err := mcp.Start(ctx, cmd)
		if err != nil {
			return fail(err)
		}
		servers = append(servers, server)
		more, err := server.Tools(ctx)
		if err != nil {
			return fail(fmt.Errorf("%s: %w", command, err))
		}
		for _, t := range more {
			if name := t.Spec().Name; names[name] {
				return fail(fmt.Errorf("%s: another tool is named %s", command, name))
			}
			names[t.Spec().Name] = true
		}
		tools = append(tools, more...)
	}
	return tools, servers, nil
}
