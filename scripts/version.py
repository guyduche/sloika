import os
import yaml
import subprocess

if os.environ.has_key('CI'):
    version_template = '%(major)s.%(minor)s.%(patch)s'
else:
    version_template = '%(major)s.%(minor)s.dev%(patch)s'

cmd = 'git rev-list --all --count'
out, err = subprocess.Popen(cmd.split(),stdout=subprocess.PIPE).communicate()
patch = int(out)

with open("version.yaml", 'r') as stream:
    try:
	D = yaml.load(stream)
	D['patch'] = patch
        version = version_template%D
    except yaml.YAMLError as exc:
        print(exc)

print(version)
