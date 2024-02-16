In order to get a working environment, you need to install our package manager, orchestra.

First of all, let's clone the orchestra configuration:

```{bash noorchestra}
$ git clone https://github.com/revng/orchestra
Cloning into 'orchestra'...
$ cd orchestra
```

Let's now install the actual package manager application, `revng-orchestra`.
<br />You can either install it globally for the current user or in a virtualenv.

```{bash notest title="Install globally for the current user"}
$ python3 -m pip install --user --force-reinstall https://github.com/revng/revng-orchestra/archive/master.zip
$ export PATH="$HOME/.local/bin:$PATH"
```

```{bash title="Install in a virtualenv" notest}
$ python -m venv venv
$ source venv/bin/activate
$ pip install --upgrade pip wheel certifi
$ pip install https://github.com/revng/revng-orchestra/archive/master.zip
```

At this point, from the `orchestra` repository, we can sync all the information with the remote repositories:

```{bash silent}
$ orc update
```

Currently, rev.ng is split in two repositories `revng`, containing infrastructure and the lifter, and `revng-c` containing the backend of the decompiler, the part actually emitting the C code.
These two repos will eventually get merged.

To install `revng`, `revng-c` and its dependencies from the binary archives run the following command:

```{bash notest}
$ orc install revng revng-c
```

Now you can enter the environment where you can use `revng`:

```{bash notest}
$ orc shell
$ revng artifact --help
```

### Building from source

In order to build from source, add the following to `.orchestra/config/user_options.yml`:

```diff
--- a/.orchestra/config/user_options.yml
+++ b/.orchestra/config/user_options.yml
@@ -11,6 +11,6 @@
 binary_archives:
   - origin: "https://rev.ng/gitlab/revng/binary-archives.git"

-#! #@overlay/replace
-#! build_from_source:
-#!   - component-name
+#@overlay/replace
+build_from_source:
+  - revng
+  - revng-c
```

We can now install and test `revng` and `revng-c`:

```{bash notest}
$ orc install --test revng revng-c
```

This will clone the sources into `sources/revng` and `sources/revng-c`, build them, install them (in `root/`) and run the test suites.
