import moviepy

aud = moviepy.editor.AudioFileClip("seventh-seal-by-kevin-macleod-from-filmmusic-io.mp3")
aud = aud.cutout(50,54)
vid = moviepy.editor.VideoFileClip("starry.mp4")
scape = vid.set_audio(aud)
scape.write_videofile("space_sound.mp4", audio_codec='aac')
aud.close()
vid.close()
scape.close()