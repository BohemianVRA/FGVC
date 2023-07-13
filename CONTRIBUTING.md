# Contributing to FGVC

## Creating Package Release

Automatic CI/CD workflows are set using GitHub Actions.
Create new git `tag` to trigger **Build and Create Release** action.

```bash
git tag -a v1.0.0 -m "Changelog:
* First change.
* Second change."
git push origin v1.0.0
```


## Improving The Documentation

The documentation webpage is generated using [Sphinx](https://www.sphinx-doc.org/en/master/index.html) tool.

Run the following commands to generate the documentation.
```bash
# install dependencies and fgvc
pip install -r requirements.txt
pip install -r requirements_dev.txt
pip install -e .

# generate documentations
cd docs
make html
```
