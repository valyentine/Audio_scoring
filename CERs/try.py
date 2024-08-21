import whisper

model = whisper.load_model("base")
result = model.transcribe("audio/dz_chushibiao.wav")
print(result["text"])