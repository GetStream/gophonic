// Copyright 2026 The gophonic authors
// SPDX-License-Identifier: BSD-2-Clause

package duplex

import (
	"context"
	"fmt"
	"reflect"
	"strings"

	"github.com/GetStream/gophonic/chat"
	"github.com/thesyncim/vibejson"
)

// Tool is a function the agent may call instead of, or as well as,
// answering: its spec tells the model what it does, and Run does it. Func
// makes one of a typed Go function.
type Tool struct {
	chat.ToolSpec
	// Run performs a call, given the model's arguments as a JSON object,
	// once the turn that asked for it is over; ctx ends if the speaker
	// interrupts. The result joins the conversation; with reply set, the
	// agent then answers again knowing it, as after a lookup, and
	// otherwise says nothing more, as after being asked to be quiet.
	Run func(ctx context.Context, arguments []byte) (result string, reply bool, err error)
}

// Func makes a Tool of a Go function whose arguments are a struct. The
// struct is the tool's parameters: each exported field is one, named by
// its json tag, described by its desc tag, limited to the comma-separated
// values of its enum tag, and optional when tagged omitempty. Strings,
// booleans, numbers, slices, and nested structs are supported; Func
// panics on anything else. After the call, the agent answers knowing its
// result; see Silent.
//
//	duplex.Func("now", "The current date and time.",
//		func(ctx context.Context, args struct {
//			Timezone string `json:"timezone,omitempty" desc:"an IANA time zone"`
//		}) (string, error) {
//			...
//		})
func Func[Args any](name, description string, run func(ctx context.Context, args Args) (string, error)) Tool {
	t := reflect.TypeFor[Args]()
	if t.Kind() != reflect.Struct {
		panic(fmt.Sprintf("duplex: tool %s: arguments are a %v, not a struct", name, t))
	}
	var schema strings.Builder
	writeSchema(&schema, t, "")
	return Tool{
		ToolSpec: chat.ToolSpec{Name: name, Description: description, Parameters: schema.String()},
		Run: func(ctx context.Context, arguments []byte) (string, bool, error) {
			var args Args
			if err := vibejson.Unmarshal(arguments, &args); err != nil {
				return "", false, fmt.Errorf("arguments %s: %w", arguments, err)
			}
			result, err := run(ctx, args)
			return result, true, err
		},
	}
}

// Silent returns t made to end the reply: after it runs, the agent says
// nothing more, as after being asked to be quiet.
func (t Tool) Silent() Tool {
	run := t.Run
	t.Run = func(ctx context.Context, arguments []byte) (string, bool, error) {
		result, _, err := run(ctx, arguments)
		return result, false, err
	}
	return t
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
		panic(fmt.Sprintf("duplex: tool arguments of type %v", t))
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
