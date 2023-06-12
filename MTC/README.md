# MedTetCleaner

## Project setup
```
npm install
```

**NOTES**: you might need to install the customized Tui Image Editor manually. For this, go to the following [project repository](https://github.com/NicoSzela/tui.image-editor/tree/master/apps/image-editor). Using npm (node version: v14.21.3), run `npm install` inside the apps/image-editor directory. Then run `npm run buid`, this will create a dist folder that you need to copy inside the tui-image-editor directory located in the node_modules directory of MTC (generated after the npm install command).

### Compiles and hot-reloads for development
```
npm run serve
```

### Compiles and minifies for production
```
npm run build
```

**NOTES**: this step will generate a dist folder. The content of this folder shoud be placed inside the OrthancPlugin/orthanc/MedTextCleaner direcotry. You will also need to modify the generated index.html file by adding `/MedTextCleaner` to the start of src and href (e.g. src="/MedTextCleaner/js/app.8b6fe163.js", href="/MedTextCleaner/css/app.1c04c876.css")

### Lints and fixes files
```
npm run lint
```
