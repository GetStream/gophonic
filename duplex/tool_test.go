// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package duplex

import (
	"context"
	"strings"
	"testing"
)

func TestFunc(t *testing.T) {
	type place struct {
		City string `json:"city"`
	}
	tool := Func("book", "Book a table.", func(_ context.Context, args struct {
		Name   string   `json:"name" desc:"who the table is for"`
		People int      `json:"people"`
		Time   string   `json:"time,omitempty" enum:"lunch, dinner"`
		Where  *place   `json:"where,omitempty"`
		Notes  []string `json:"notes,omitempty"`
		hidden int
	}) (string, error) {
		return args.Name + " " + strings.Repeat("x", args.People) + " " + args.Time + " " + args.Where.City + " " + strings.Join(args.Notes, "+"), nil
	})
	want := `{"type": "object", "properties": {` +
		`"name": {"type": "string", "description": "who the table is for"}, ` +
		`"people": {"type": "integer"}, ` +
		`"time": {"type": "string", "enum": ["lunch", "dinner"]}, ` +
		`"where": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}, ` +
		`"notes": {"type": "array", "items": {"type": "string"}}}, ` +
		`"required": ["name", "people"]}`
	if tool.Parameters != want {
		t.Fatalf("schema\n%s\nwant\n%s", tool.Parameters, want)
	}
	got, reply, err := tool.Run(context.Background(), []byte(`{"name": "Ana", "people": 3, "time": "dinner", "where": {"city": "Porto"}, "notes": ["a", "b"]}`))
	if err != nil || !reply || got != "Ana xxx dinner Porto a+b" {
		t.Fatalf("%q %v %v", got, reply, err)
	}
	if _, reply, _ := tool.Silent().Run(context.Background(), []byte(`{"name": "Ana", "people": 1, "where": {}}`)); reply {
		t.Fatal("a silent tool asked for a reply")
	}
	if _, _, err := tool.Run(context.Background(), []byte(`{"people": "many"}`)); err == nil {
		t.Fatal("bad arguments ran")
	}
}
