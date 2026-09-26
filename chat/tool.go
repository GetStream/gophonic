// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package chat

import (
	"context"
	"fmt"
	"reflect"
	"strings"

	"github.com/thesyncim/vibejson"
)

// Func makes a Tool of a Go function whose arguments are a struct. The
// struct is the tool's parameters: each exported field is one, named by
// its json tag, described by its desc tag, limited to the comma-separated
// values of its enum tag, and optional when tagged omitempty. Strings,
// booleans, numbers, slices, and nested structs are supported; Func
// panics on anything else.
//
//	chat.Func("now", "The current date and time.",
//		func(ctx context.Context, args struct {
//			Timezone string `json:"timezone,omitempty" desc:"an IANA time zone"`
//		}) (string, error) {
//			...
//		})
func Func[Args any](name, description string, run func(ctx context.Context, args Args) (string, error)) Tool {
	t := reflect.TypeFor[Args]()
	if t.Kind() != reflect.Struct {
		panic(fmt.Sprintf("chat: tool %s: arguments are a %v, not a struct", name, t))
	}
	var schema strings.Builder
	writeSchema(&schema, t, "")
	return &funcTool[Args]{spec: ToolSpec{Name: name, Description: description, Parameters: schema.String()}, run: run}
}

// funcTool is the Tool of a typed function.
type funcTool[Args any] struct {
	spec ToolSpec
	run  func(ctx context.Context, args Args) (string, error)
}

func (t *funcTool[Args]) Spec() ToolSpec { return t.spec }

func (t *funcTool[Args]) Call(ctx context.Context, arguments []byte) (string, error) {
	var args Args
	if err := vibejson.Unmarshal(arguments, &args); err != nil {
		return "", fmt.Errorf("arguments %s: %w", arguments, err)
	}
	return t.run(ctx, args)
}

// writeSchema writes the JSON Schema of t, described by desc.
func writeSchema(b *strings.Builder, t reflect.Type, desc string) {
	for t.Kind() == reflect.Pointer {
		t = t.Elem()
	}
	b.WriteString(`{"type": `)
	switch t.Kind() {
	case reflect.String:
		b.WriteString(`"string"`)
	case reflect.Bool:
		b.WriteString(`"boolean"`)
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64,
		reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		b.WriteString(`"integer"`)
	case reflect.Float32, reflect.Float64:
		b.WriteString(`"number"`)
	case reflect.Slice, reflect.Array:
		b.WriteString(`"array", "items": `)
		writeSchema(b, t.Elem(), "")
	case reflect.Struct:
		b.WriteString(`"object", "properties": {`)
		var required []string
		n := 0
		for i := range t.NumField() {
			f := t.Field(i)
			name, opts, _ := strings.Cut(f.Tag.Get("json"), ",")
			if !f.IsExported() || name == "-" {
				continue
			}
			if name == "" {
				name = f.Name
			}
			if n++; n > 1 {
				b.WriteString(", ")
			}
			b.WriteString(quote(name) + ": ")
			field := strings.Builder{}
			writeSchema(&field, f.Type, f.Tag.Get("desc"))
			s := field.String()
			if enum := f.Tag.Get("enum"); enum != "" {
				var vals []string
				for v := range strings.SplitSeq(enum, ",") {
					vals = append(vals, quote(strings.TrimSpace(v)))
				}
				s = s[:len(s)-1] + `, "enum": [` + strings.Join(vals, ", ") + "]}"
			}
			b.WriteString(s)
			if !strings.Contains(opts, "omitempty") {
				required = append(required, quote(name))
			}
		}
		b.WriteString("}")
		if len(required) > 0 {
			b.WriteString(`, "required": [` + strings.Join(required, ", ") + "]")
		}
	default:
		panic(fmt.Sprintf("chat: tool arguments of type %v", t))
	}
	if desc != "" {
		b.WriteString(`, "description": ` + quote(desc))
	}
	b.WriteString("}")
}

// quote returns s as a JSON string.
func quote(s string) string {
	q, err := vibejson.Marshal(&s)
	if err != nil {
		panic(err)
	}
	return string(q)
}
