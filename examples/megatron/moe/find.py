# export MEGATRON_LM_PATH='/usr/local/lib/python3.11/site-packages/megatron' 

import site
package_paths = site.getsitepackages()
for path in package_paths:
    print(path)