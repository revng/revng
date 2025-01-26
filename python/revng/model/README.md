# rev.ng model classes

By default, importing the top level package will expose the classes for the latest version of the model.

Example: deserializing a model

```python
import yaml
from revng import model as m

with open("/path/to/model.yaml") as f:
    model = yaml.load(f, Loader=m.YamlLoader)
```

It is not possible to import a model with a schema version other than the latest.
