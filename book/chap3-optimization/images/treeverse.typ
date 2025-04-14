#import "@preview/cetz:0.2.2": canvas, draw, tree, plot
#set page(width: auto, height: auto, margin: 5pt)

// Function to draw a single tape state
#let show-tape(n, checkpoints, removed, new, y, ngrad, unit) = {
    import draw: *
    // Draw grid cells
    for i in range(n - ngrad + 1) {
      rect(((i + 0.5) * unit - 0.4 * unit, y - 0.4 * unit), ((i + 0.5) * unit + 0.4 * unit, y + 0.4 * unit), fill: rgb(190, 190, 190), stroke: none)
    }
    
    // Draw gradient cells
    for i in range(n - ngrad + 1, n) {
      rect(((i + 0.5) * unit - 0.4 * unit, y - 0.4 * unit), ((i + 0.5) * unit + 0.4 * unit, y + 0.4 * unit), fill: red, stroke: none)
    }
    
    // Draw removed pebbles
    for p in removed.filter(p => not checkpoints.contains(p)) {
      circle(((p + 0.5) * unit, y), radius: 0.25 * unit, stroke: (paint: black, thickness: 0.5pt), fill: none)
    }
    
    // Draw new pebbles
    for p in new {
      circle(((p + 0.5) * unit, y), radius: 0.25 * unit, fill: black)
    }
    
    // Draw existing pebbles
    for p in checkpoints.filter(p =>not new.contains(p)) {
      circle(((p + 0.5) * unit, y), radius: 0.25 * unit, fill: rgb(85, 85, 85), stroke: rgb(85, 85, 85))
    }
  }
 
#let visualize-treeverse(filename) = {
    import draw: *
    // Load and parse the JSON data
    let data = json(filename)
    let actions = data.actions
    
    // Calculate dimensions
    let N = 30  // From the filename treeverse-30-5.json
    let call-count = actions.filter(a => a.action == "call").len() + 1
    
    // // Setup canvas
    let width = 12
    let height = call-count * 0.5
    let unit = width / (N + 1)
    
    // Colors
    let bg-color = rgb(170, 170, 170)
    let new-color = black
    let exist-color = rgb(85, 85, 85)
    let grad-color = red
    
    // Draw the visualization
    let y = 0.5
    let checkpoints = (0,)
    let removed = ()
    let new = ()
    let ngrad = 1
    
    // // Draw grid background first
    // for i in range(N) {
    //   rect(((i + 0.5) * unit - 0.4 * unit, height - 0.5 - 0.4 * unit), ((i + 0.5) * unit + 0.4 * unit, height - 0.5 + 0.4 * unit), fill: bg-color, stroke: none)
    // }
    
    // Process actions
    for (i, act) in actions.enumerate() {
      let pebbles = checkpoints
      if act.action == "call" {
        // Add new checkpoint
        pebbles.push(act.step + 1)
        removed.push(act.step)
        
        // Draw the state
        new = (act.step+1,)
      } else if act.action == "store" {
        // Store checkpoint
        checkpoints.push(act.step)
        continue
      } else if act.action == "fetch" {
        // Remove checkpoint
        pebbles.push(act.step)
        checkpoints = checkpoints.filter(p => p != act.step)
        removed.push(act.step)
        continue
      } else if act.action == "grad" {
        ngrad += 1
        removed.push(act.step)
        
        // If last action, draw the state
        if i != actions.len() - 1 {
          continue
        }
      }
      show-tape(N, pebbles, removed, new, -y, ngrad, unit)
      removed = ()
      y += 0.4
    }
  }

#figure(canvas({
  import draw: *
  // Function to visualize treeverse log
 
  // Visualize the treeverse log
  visualize-treeverse("treeverse-30-5.json")
  //content((), [#res])
}))