from pathlib import Path
import json,hashlib
repo=Path('/Users/thesyncim/Documents/ChatGPT/goinfer')
out=Path('/private/tmp/goinfer-singlecore-profile')
out.mkdir(parents=True,exist_ok=True)
repl={}
hashes={}
def get(name):
 p=repo/'whisper'/name
 data=p.read_text(); hashes[str(p)]=hashlib.sha256(data.encode()).hexdigest()
 return data.replace('import (','import (\n "time"',1)
def save(name,s):
 p=out/name;p.write_text(s);repl[str(repo/'whisper'/name)]=str(p)
def mark(s,old,new):
 assert s.count(old)==1, (old,s.count(old))
 return s.replace(old,new,1)
def tick(var,idx): return f'\n singleCoreStages[{idx}] += time.Since({var})\n {var}=time.Now()\n'
s=get('transcriber.go')
s=mark(s,'\tif err := FeaturesInto(pcm, t.mel, t.frontend);',' windowStageStart:=time.Now()\n\tif err := FeaturesInto(pcm, t.mel, t.frontend);')
for needle,idx in [('\tif err := t.model.EncodeInto(t.mel, t.audio, t.encoder);',0),('\tif err := t.model.BeginDecode(t.audio, t.decoder);',1),('\tpromptLen, err := t.policy.PromptInto(t.tokens[:0], nil, nil)',2),('\tstart := len(dst)',3)]: s=mark(s,needle,tick('windowStageStart',idx)+needle)
save('transcriber.go',s)
s=get('encoder.go')
s=mark(s,'\t// Whisper\'s audio stem is Conv1d',' encoderStageStart:=time.Now()\n\t// Whisper\'s audio stem is Conv1d')
s=mark(s,'\tfor i := 0; i < AudioLayers; i++ {\n\t\tif err := encodeBlock',tick('encoderStageStart',4)+'\tfor i := 0; i < AudioLayers; i++ {\n\t\tif err := encodeBlock')
s=mark(s,'\tfor t := 0; t < AudioFrames; t++ {\n\t\tstart := t * AudioState\n\t\trow := dst[start : start+AudioState]\n\t\tlayerNormRow(row, row, weights.finalNormW, weights.finalNormB)', '\n encoderStageStart=time.Now()\n\tfor t := 0; t < AudioFrames; t++ {\n\t\tstart := t * AudioState\n\t\trow := dst[start : start+AudioState]\n\t\tlayerNormRow(row, row, weights.finalNormW, weights.finalNormB)')
s=mark(s,'\tif trace != nil {\n\t\ttrace(AudioLayers+2, dst)',tick('encoderStageStart',5)+'\tif trace != nil {\n\t\ttrace(AudioLayers+2, dst)')
s=mark(s,'func encodeBlock(dst []float32, block encoderBlockWeights, packed packedEncoderBlock, w *EncoderWorkspace) error {','func encodeBlock(dst []float32, block encoderBlockWeights, packed packedEncoderBlock, w *EncoderWorkspace) error {\n blockStageStart:=time.Now()')
for needle,idx in [('\tif err := w.gemm.Mul(packed.query,',6),('\t// Each query tile consumes',7),('\tif err := w.gemm.Mul(packed.out,',8),('\tfor t := 0; t < AudioFrames; t++ {\n\t\tstart := t * AudioState\n\t\trow := dst[start : start+AudioState]\n\t\tlayerNormRow(row, w.normalized[start:start+AudioState], block.mlpNormW, block.mlpNormB)',9),('\tif err := w.gemm.Mul(packed.mlpIn,',10),('\tif err := w.activate(w.feedForward,',11),('\tif err := w.gemm.Mul(packed.mlpOut,',12),('\taddRowBias(w.normalized, block.mlpOutB,',13),('\taddInPlace(dst, w.normalized)\n\treturn nil',14)]:
 if idx==14:s=mark(s,needle,'\taddInPlace(dst, w.normalized)'+tick('blockStageStart',14)+'\treturn nil')
 else:s=mark(s,needle,tick('blockStageStart',idx)+needle)
save('encoder.go',s)
s=get('encoder_attention.go')
s=mark(s,'\theadSize := a.state / a.heads',' attnStageStart:=time.Now()\n\theadSize := a.state / a.heads')
s=mark(s,'\ta.q, a.dst = q, dst',tick('attnStageStart',15)+'\ta.q, a.dst = q, dst')
s=mark(s,'\t\t\tif err := a.keys[head].Mul',' tileStageStart:=time.Now()\n\t\t\tif err := a.keys[head].Mul')
s=mark(s,'\t\t\tsoftmaxRows(scores, rows, a.rows)',tick('tileStageStart',16)+'\t\t\tsoftmaxRows(scores, rows, a.rows)'+tick('tileStageStart',17))
s=mark(s,'\t\t\tif err := a.values[head].Mul(a.dst[offset:], a.state, scores, a.rows, rows); err != nil {\n\t\t\t\ta.errors[worker] = err\n\t\t\t\treturn\n\t\t\t}', '\t\t\tif err := a.values[head].Mul(a.dst[offset:], a.state, scores, a.rows, rows); err != nil {\n\t\t\t\ta.errors[worker] = err\n\t\t\t\treturn\n\t\t\t}'+tick('tileStageStart',18))
save('encoder_attention.go',s)
s=get('decoder.go')
s=mark(s,'\t\t// Residual self attention:', ' decoderStageStart:=time.Now()\n\t\t// Residual self attention:')
s=mark(s,'\t\t// Residual cross attention:',tick('decoderStageStart',19)+'\t\t// Residual cross attention:')
s=mark(s,'\t\t// Residual MLP:',tick('decoderStageStart',20)+'\t\t// Residual MLP:')
s=mark(s,'\t\tlinearInto(s.projected, s.mlp, lw.mlpOut.weight, lw.mlpOut.bias, 4*state, state)\n\t\taddInto(s.x, s.projected)','\t\tlinearInto(s.projected, s.mlp, lw.mlpOut.weight, lw.mlpOut.bias, 4*state, state)\n\t\taddInto(s.x, s.projected)'+tick('decoderStageStart',21))
s=mark(s,'\tif s.gemm == nil {\n\t\tif err := whispergemm.MulVector(logits,',' vocabStageStart:=time.Now()\n\tif s.gemm == nil {\n\t\tif err := whispergemm.MulVector(logits,')
s=mark(s,'\ts.nextPos++',tick('vocabStageStart',22)+'\ts.nextPos++')
s=mark(s,'\t\tif err := s.multiply(s.crossKey[layer],',' crossStageStart:=time.Now()\n\t\tif err := s.multiply(s.crossKey[layer],')
s=mark(s,'\t\tbias := s.weights.layers[layer].crossV.bias',tick('crossStageStart',23)+'\t\tbias := s.weights.layers[layer].crossV.bias')
s=mark(s,'\t\tfor i := range keys {\n\t\t\tkeys[i] *= scale\n\t\t}', '\t\tfor i := range keys {\n\t\t\tkeys[i] *= scale\n\t\t}'+tick('crossStageStart',24))
save('decoder.go',s)
test='''package whisper
import("testing";"time";"os";"runtime";"runtime/pprof";"encoding/json";"slices";"path/filepath")
var singleCoreStages [25]time.Duration
func TestSingleCoreStageProfile(t *testing.T) {
 runtime.GOMAXPROCS(1)
 m,err:=Load("/private/tmp/gophonic-tiny.en.gophonic");if err!=nil{t.Fatal(err)}
 w,err:=NewTranscriberWithWorkers(m,1);if err!=nil{t.Fatal(err)};defer w.Close()
 info,err:=os.Stat("../testdata/whisper_jfk.pcm.f32le");if err!=nil{t.Fatal(err)}
 pcm,err:=readFloatFixture("../testdata/whisper_jfk.pcm.f32le",int(info.Size()/4));if err!=nil{t.Fatal(err)}
 data,err:=os.ReadFile("../testdata/whisper/jfk.oracle.json");if err!=nil{t.Fatal(err)}
 var oracle struct {Transcript string `json:"transcript"`;Prefix []int `json:"prefix"`;Tokens []int `json:"tokens"`};if err:=json.Unmarshal(data,&oracle);err!=nil{t.Fatal(err)}
 expected:=append(append(slices.Clone(oracle.Prefix),oracle.Tokens...),50256)
 dst:=make([]byte,0,2048)
 type sample struct{Total time.Duration `json:"total_ns"`;Stages [25]time.Duration `json:"stages_ns"`}
 names:=[]string{"frontend","encoder","cross_kv","tokens","encoder_stem","encoder_final_norm","encoder_attn_norm","encoder_qkv","encoder_attention","encoder_out","encoder_mlp_norm","encoder_mlp_expand","encoder_mlp_gelu","encoder_mlp_contract","encoder_mlp_residual","attention_pack","attention_qk","attention_softmax","attention_pv","decoder_self","decoder_cross","decoder_mlp","decoder_vocabulary","cross_kv_matmul","cross_kv_layout"}
 results:=make([]sample,0,16)
 outDir:="/private/tmp/goinfer-singlecore-profile"
 for i:=-5;i<16;i++ {
  if i==0 {f,err:=os.Create(filepath.Join(outDir,"cpu.pprof"));if err!=nil{t.Fatal(err)};defer f.Close();if err:=pprof.StartCPUProfile(f);err!=nil{t.Fatal(err)}}
  singleCoreStages=[25]time.Duration{};start:=time.Now();text,err:=w.TranscribeWindowInto(pcm,dst);elapsed:=time.Since(start)
  if err!=nil{t.Fatal(err)};if string(text)!=oracle.Transcript||!slices.Equal(w.tokens,expected){t.Fatalf("wrong oracle text=%q tokens=%v",text,w.tokens)}
  if i>=0 {results=append(results,sample{elapsed,singleCoreStages})}
 }
 pprof.StopCPUProfile()
 f,err:=os.Create(filepath.Join(outDir,"go-stage-samples.json"));if err!=nil{t.Fatal(err)};defer f.Close()
 report:=struct{Toolchain string `json:"toolchain"`;GOARCH string `json:"goarch"`;GOMAXPROCS int `json:"gomaxprocs"`;Workers int `json:"workers"`;Warmups int `json:"warmups"`;Tokens []int `json:"tokens_including_prefix_eos"`;Names []string `json:"stage_names"`;Samples []sample `json:"samples"`}{runtime.Version(),runtime.GOARCH,runtime.GOMAXPROCS(0),1,5,expected,names,results}
 if err:=json.NewEncoder(f).Encode(report);err!=nil{t.Fatal(err)}
}
'''
save('singlecore_profile_experiment_test.go',test)
(out/'overlay.json').write_text(json.dumps({'Replace':repl},indent=2))
(out/'source-sha256.json').write_text(json.dumps(hashes,indent=2))
