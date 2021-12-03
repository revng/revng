# rev.ng model classes

By default, importing the top level package will expose the classes for the latest version of the model.

Example: deserializing a model

```python
import yaml
from revng import model as m

with open("/path/to/model.yaml") as f:
    serialized_model = yaml.load(f)

model = m.Binary.parse_obj(serialized_model)
```

If you need to access a specific version of the model you can import it like so:

```python
from revng.model import v1

v1.Binary.parse_obj(...)
```
