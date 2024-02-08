## Binary distribution

If you have access to a binary release of rev.ng just run the following command:

```{bash notest}
$ tar xaf revng-*.tar.gz
$ cd revng
$ export PATH="$PWD:$PATH"
$ revng artifact --help
```

## Using orchestra

In order to build from source, you need to install our package manager, orchestra.

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

To install `revng` and its dependencies from the binary archives run the following command:

```{bash notest}
$ orc install revng
```

Now you can enter the environment where you can use `revng`:

```{bash notest}
$ orc shell
$ revng artifact --help
$ cat .orchestra/config/user_options.yml
```

### Building from source

Add the following to `.orchestra/config/user_options.yml`:

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
```

We can now install and test `revng`:

```{bash notest}
$ orc install --test revng
```

This will clone the sources into `sources/revng`, build it, install it (in `root/`) and run the test suite.

### rev.ng Developers

If you have access to the private rev.ng repositories (mainly, `revng-c`), you can build from source that too:

```diff
--- a/.orchestra/config/user_options.yml
+++ b/.orchestra/config/user_options.yml
@@ -14,3 +14,4 @@
 #@overlay/replace
 build_from_source:
   - revng
+  - revng-c
```

```{bash notest}
$ orc install --test revng-c
```
