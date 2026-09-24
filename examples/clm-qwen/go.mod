module github.com/GetStream/gophonic/examples/clm-qwen

go 1.27.0

require (
	github.com/GetStream/gophonic v0.0.0
	github.com/townsendmerino/aikit v1.47.0
	github.com/townsendmerino/goinfer v0.19.1-0.20260924101646-2f2b429898b2
	golang.org/x/text v0.40.0
)

require golang.org/x/sys v0.47.0 // indirect

replace github.com/GetStream/gophonic => ../..
