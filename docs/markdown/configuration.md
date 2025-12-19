# Settings

Settings are managed in the FM2PROF Configuration file

```python exec="on"
import json

with open('fm2prof/configurationfile_template.json', 'r') as f:
    config = json.load(f)
    for section_name, section_content in config["sections"].items():
        print(f"## " + section_name.capitalize())
        for key_value, key_content in section_content.items():
            print (f"### {key_value}")
            print (f"""

{key_content.get('hint')}

| Input   |      Value      |
|----------|:-------------:|
| Type |  {key_content.get('type')}  |
| Default value |{key_content.get('value')}|

""")


```