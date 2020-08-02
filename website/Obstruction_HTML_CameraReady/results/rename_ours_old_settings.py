import glob
import os

ours = sorted(glob.glob('reflection/**/*_Ours.png'))
ours_old_settings = sorted(glob.glob('reflection/**/*_Ours_oldSetting.png'))

for o in ours:
    os.remove(o)

for o in ours_old_settings:
    os.rename(o, o.replace('_Ours_oldSetting.png', '_Ours.png'))

