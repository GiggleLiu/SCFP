// Images are created by Brave Workz, from the Line Store:
// https://store.line.me/emojishop/product/61075ddc3cc24041bb144d70/en
// https://store.line.me/emojishop/product/5f5722aaf5a60e01d42e398a/en

// We only use the images for non-commercial purposes.
// If the author wants to remove these images, please contact us.

#let myimage(path, size, inline, offset) = if inline {box(image(path, height: size), baseline: (size - 20pt)/2 + offset)} else {image(path, height: size)}
#let myimage1(path, size, inline, offset) = if inline {box(image(path, height: size - 6pt), inset: 3pt, baseline: (size - 17pt)/2 + offset)} else {image(path, height: size - 10pt)}
#let bob(size: 40pt, inline: true, offset: 0pt) = myimage("images/pixel2/001.png", size, inline, offset)
#let alice(size: 40pt, inline: true, offset: 0pt) = myimage("images/pixel2/002.png", size, inline, offset)
#let christina(size: 40pt, inline: true, offset: 0pt) = myimage("images/pixel2/003.png", size, inline, offset)
#let mary(size: 40pt, inline: true, offset: 0pt) = myimage("images/pixel2/004.png", size, inline, offset)
#let eve(size: 40pt, inline: true, offset: 0pt) = myimage("images/pixel2/005.png", size, inline, offset)
#let frank(size: 40pt, inline: true, offset: 0pt) = myimage("images/pixel1/010.png", size, inline, offset)
#let grace(size: 40pt, inline: true, offset: 0pt) = myimage("images/pixel2/007.png", size, inline, offset)
#let henry(size: 40pt, inline: true, offset: 0pt) = myimage("images/pixel2/008.png", size, inline, offset)
#let ivan(size: 40pt, inline: true, offset: 0pt) = myimage("images/pixel1/007.png", size, inline, offset)
#let william(size: 40pt, inline: true, offset: 0pt) = myimage1("images/pixel1/011.png", size, inline, offset)
#let kate(size: 40pt, inline: true, offset: 0pt) = myimage("images/pixel2/011.png", size, inline, offset)
#let linda(size: 40pt, inline: true, offset: 0pt) = myimage("images/pixel2/012.png", size, inline, offset)
#let david(size: 40pt, inline: true, offset: 0pt) = myimage("images/pixel1/013.png", size, inline, offset)
#let nancy(size: 40pt, inline: true, offset: 0pt) = myimage("images/pixel2/014.png", size, inline, offset)
#let olivia(size: 40pt, inline: true, offset: 0pt) = myimage("images/pixel2/015.png", size, inline, offset)
#let patrick(size: 40pt, inline: true, offset: 0pt) = myimage("images/pixel1/018.png", size, inline, offset)
#let murphy(size: 40pt, inline: true, offset: 0pt) = myimage("images/pixel2/016.png", size, inline, offset)
#let rachel(size: 40pt, inline: true, offset: 0pt) = myimage("images/pixel2/018.png", size, inline, offset)
#let sam(size: 40pt, inline: true, offset: 0pt) = myimage("images/pixel1/024.png", size, inline, offset)
#let tom(size: 40pt, inline: true, offset: 0pt) = myimage("images/pixel1/012.png", size, inline, offset)
#let ina(size: 40pt, inline: true, offset: 0pt) = myimage("images/pixel2/021.png", size, inline, offset)
#let victor(size: 40pt, inline: true, offset: 0pt) = myimage("images/pixel1/016.png", size, inline, offset)
#let wendy(size: 40pt, inline: true, offset: 0pt) = myimage("images/pixel2/023.png", size, inline, offset)
#let xavier(size: 40pt, inline: true, offset: 0pt) = myimage("images/pixel1/014.png", size, inline, offset)
#let mario(size: 40pt, inline: true, offset: 0pt) = myimage1("images/pixel1/004.png", size, inline, offset)
#let zoe(size: 40pt, inline: true, offset: 0pt) = myimage("images/pixel2/026.png", size, inline, offset)
#let amanda(size: 40pt, inline: true, offset: 0pt) = myimage1("images/pixel1/021.png", size, inline, offset)
#let brian(size: 40pt, inline: true, offset: 0pt) = myimage1("images/pixel1/002.png", size, inline, offset)
#let carol(size: 40pt, inline: true, offset: 0pt) = myimage("images/pixel2/029.png", size, inline, offset)
#let daniel(size: 40pt, inline: true, offset: 0pt) = myimage("images/pixel2/030.png", size, inline, offset)
#let elizabeth(size: 40pt, inline: true, offset: 0pt) = myimage("images/pixel2/031.png", size, inline, offset)
#let felix(size: 40pt, inline: true, offset: 0pt) = myimage1("images/pixel1/003.png", size, inline, offset)
#let gina(size: 40pt, inline: true, offset: 0pt) = myimage("images/pixel2/033.png", size, inline, offset)
#let henry(size: 40pt, inline: true, offset: 0pt) = myimage("images/pixel2/034.png", size, inline, offset)
#let isabelle(size: 40pt, inline: true, offset: 0pt) = myimage("images/pixel2/035.png", size, inline, offset)
#let james(size: 40pt, inline: true, offset: 0pt) = myimage1("images/pixel1/001.png", size, inline, offset)
#let kate(size: 40pt, inline: true, offset: 0pt) = myimage("images/pixel2/037.png", size, inline, offset)
#let linda(size: 40pt, inline: true, offset: 0pt) = myimage("images/pixel2/038.png", size, inline, offset)
#let mary(size: 40pt, inline: true, offset: 0pt) = myimage("images/pixel2/039.png", size, inline, offset)
#let nancy(size: 40pt, inline: true, offset: 0pt) = myimage("images/pixel2/040.png", size, inline, offset)


#alice() Alice
#bob() Bob
#christina() Christina
#david() David
#eve() Eve
#frank() Frank
#grace() Grace
#henry() Henry
#ivan() Ivan
#james() James
#kate() Kate
#linda() Linda
#mary() Mary
#nancy() Nancy
#olivia() Olivia
#patrick() Patrick
#murphy() Murphy
#rachel() Rachel
#sam() Sam
#tom() Tom
#ina() Ina
#victor() Victor
#wendy() Wendy
#xavier() Xavier
#mario() Mario
#zoe() Zoe
#amanda() Amanda
#brian() Brian
#carol() Carol
#daniel() Daniel
#elizabeth() Elizabeth
#felix() Felix
#gina() Gina
#henry() Henry
#isabelle() Isabelle
#william() William
#kate() Kate
#linda() Linda
#mary() Mary
#nancy() Nancy
