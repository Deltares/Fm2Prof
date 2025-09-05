# FM2PROF configuration 

```python exec="on"
import json

with open('fm2prof/configurationfile_template.json', 'r') as f:
    config = json.load(f)
    for section_name, section_content in config["sections"].items():
        print(f"## " + section_name.capitalize())
        for key_value, key_content in section_content.items():
            print (f"### {key_value}")
            print (f"""
**type**: {key_content.get('type')}

**default value**: {key_content.get('value')}

{key_content.get('hint')}
""")


```