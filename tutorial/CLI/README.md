# Data Folder for the CLI Tutorial

Install the DrugEx package along with [QSPRPred](https://github.com/CDDLeiden/QSPRPred). In order to obtain tutorial data, run the following command in this directory (/CLI):

```bash
drugex download -o examples # or 'python -m drugex.download -o examples' if you are on Windows
```

This will create an `examples` folder with pretrained models and example data in the current directory. It will also provide you with your own version of the Papyrus data set in `data/.Papyrus` so you can extract more data later if you choose so (see [Papyrus-scripts](https://github.com/OlivierBeq/Papyrus-scripts)). You can then follow the [CLI tutorial](https://cddleiden.github.io/DrugEx/docs/use.html) on the documentation page.
