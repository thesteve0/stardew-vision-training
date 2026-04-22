---
name: Synthetic screenshot generation approach
description: Learned preferences for generating synthetic Stardew Valley screenshots - clean backgrounds, accurate fonts, annotation-based positioning
type: feedback
originSessionId: dec8499b-4c1c-4f22-9d17-1a3014bb1b28
---
Use clean background + on-the-fly overlay approach for synthetic screenshots, NOT in-place text replacement (which causes ghosting/bleeding artifacts).

**Why:** In-place clearing of text regions is unreliable — edge detection fails across different screenshot resolutions, atmospheric colors (lavender rain, night), and toolbar overlap. The clean-background approach eliminates these problems entirely.

**How to apply:**
- Create clean backgrounds by removing UI elements from real screenshots
- Generate UI elements (dialog boxes, notification bubbles) from scratch and overlay at recorded positions
- For positioning, use human-annotated bounding boxes (green=#00FF00 for outer box, magenta=#FF00FF for inner elements) rather than auto-detection
- TV dialog: SmallFont at scale=2 with 1px drop shadow (text color at ~47% alpha), box anchored to toolbar left edge
- Pierre's shop: SmallFont at scale=1 with 1px drop shadow
- Always verify package safety after installing dependencies (check rocm-provided.txt + excluded-dependencies)
- Save test images to tmp/ in the project workspace (not /tmp/) so user can view them in IDE
