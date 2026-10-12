# GitHub repository and releases

The prepared source archive contains the application, documentation, images,
synthetic example, tests, workflow, citation, and license. It excludes `.git`,
virtual environments, caches, old archives, experimental VGD files, and local
analysis sessions.

## Repository presentation

Source repository: [AliAfruz/XPS-Master](https://github.com/AliAfruz/XPS-Master).
Suggested description:

> Reproducible XPS analysis with constrained fitting, model evidence, a
> multi-spectrum workspace, and publication figures. By Ali Afruz.

Set the repository website to **https://chemitro.ir**. Suggested topics:
`xps`, `spectroscopy`, `peak-fitting`, `surface-analysis`, `materials-science`,
`scipy`, `pyside6`, and `reproducible-research`.

## Work from the published source

```shell
git clone https://github.com/AliAfruz/XPS-Master.git
cd XPS-Master
```

Follow the [README installation instructions](../README.md), and run
`python -m pytest -q` before committing changes.

## Initialize an empty repository from the archive

This section applies only to an empty destination repository. For subsequent
work on the published repository, clone its existing history as shown above.

Unzip the source archive. Open a terminal in its `XPS-master` folder.
The destination is [AliAfruz/XPS-Master](https://github.com/AliAfruz/XPS-Master).

```shell
git init
git branch -M main
git config user.name "Ali Afruz"
git config user.email "a.a.afruz@gmail.com"
git add .
git commit -m "Release XPS-master 0.2.0"
git remote add origin https://github.com/AliAfruz/XPS-Master.git
git push -u origin main
```

These identity settings apply to this repository. Authenticate using GitHub's
normal Git credential flow. No password or token belongs in a source file.

## Upload through the browser

Unzip the archive first. Use GitHub's **Add file → Upload files** to upload the
contents of the project folder, including the `.github` directory and root
configuration files. Upload in multiple batches if needed. Uploading the ZIP
itself does not populate the repository with source files.

See [GitHub's file-upload instructions](https://docs.github.com/en/repositories/working-with-files/managing-files/adding-a-file-to-a-repository).

## Finish the repository page

Check that the README images render and that the Actions validation completes.
The repository URL is included in `pyproject.toml` and `CITATION.cff`.
The citation file enables GitHub's citation display; see
[GitHub's citation documentation](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-citation-files).

To publish a release, use tag **`v0.2.0`**, title **XPS-master 0.2.0**, and
the corresponding section of `CHANGELOG.md`. Attach the prepared source ZIP
and optionally the Python wheel and source distribution. These are Python
distributions, not standalone Windows installers.

For the later DOI release, follow the [Zenodo release guide](zenodo_release.md)
before publishing the GitHub release.
