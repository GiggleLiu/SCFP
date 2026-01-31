---
name: generate-learning-sheet
description: Generate a learning sheet for a specified chapter from SCFP materials
arguments: chapter_name
---

# Generate Learning Sheet

Generate a comprehensive learning sheet for the specified chapter from Scientific Computing For Physicists (SCFP).

## Pre-Work (Before Spawning Agents)

**IMPORTANT**: Before generating, you MUST:

1. **Check existing materials** - Read the source chapter file in `book/chap*/{chapter_name}.typ`
2. **Check if learning sheet exists** - Check `learning-sheets/{chapter_name}.typ` if it exists. If already good, skip generation.
3. **Understand chapter dependencies** - Read related chapters to ensure proper context
4. **Discuss with user** - Confirm the scope and focus areas:
   - What are the key learning objectives?
   - Any specific exercises or topics to emphasize?
   - Target audience level (beginner/intermediate/advanced)?

## Course Structure Reference

Refer to `syllabus/syllabus.typ` for the complete course structure, which includes:
- 13-week course arrangement with 4 modules
- Learning objectives for each week
- Chapter mappings and prerequisites
- Assessment structure

The syllabus organizes chapters into:
- **Module 1 (Weeks 1-3):** Programming Foundations
- **Module 2 (Weeks 4-6):** Numerical Linear Algebra
- **Module 3 (Weeks 7-9):** Optimization
- **Module 4 (Weeks 10-12):** Advanced Topics
- **Week 13:** Capstone Projects

## Workflow

1. **Pre-work checks** (see above)
2. **Writer Agent** generates the learning sheet
3. **Reviewer Agent** validates content accuracy and pedagogy
4. **Verification Agent** fact-checks and verifies code examples
5. Iterate until ALL agents APPROVE (max 3 cycles)
6. Compile the learning sheet to verify it works
7. Commit and push to git

## Step 1: Spawn Writer Agent

Use the Task tool to spawn the Writer agent:

```
Prompt: |
  You are a learning materials writer for SCFP (Scientific Computing For Physicists).

  **Task**: Generate a learning sheet for chapter: {chapter_name}

  **Inputs to read first**:
  1. `syllabus/syllabus.typ` - Find the week containing {chapter_name}, extract learning objectives
  2. Source chapter: Find and read the .typ file for {chapter_name} in book/chap*/
  3. `book/book.typ` - Understand the chapter's place in the curriculum
  4. Related chapters - Read prerequisite chapters as specified in syllabus
  5. `learning-sheets/` - Check existing learning sheets for style reference

  **IMPORTANT**: Use the learning objectives from the syllabus as the basis for the learning sheet.
  The syllabus contains detailed weekly objectives that should be reflected in the output.

  **Learning Sheet Structure**:
  The output file should contain these sections:

  ### 1. Learning Objectives
  - Copy the learning objectives from `syllabus/syllabus.typ` for the relevant week
  - These are already well-defined with action verbs (understand, implement, apply, analyze)
  - Add any additional objectives specific to this learning sheet if needed

  ### 2. Prerequisites
  - List required knowledge from previous chapters
  - Link to relevant prerequisite materials

  ### 3. Key Concepts Summary
  - Concise explanations of main concepts (NOT just copy from chapter)
  - Visual aids where helpful (diagrams, tables)
  - Mathematical formulas with clear notation explanations
  - Real-world analogies to aid understanding

  ### 4. Code Exercises (3-5 exercises)
  For each exercise:
  - **Problem statement** - Clear description of what to implement
  - **Hints** - Guided hints (collapsible/hidden if possible)
  - **Starter code** - Template to begin with
  - **Expected output** - What correct solution should produce
  - **Solution** - Complete working solution with comments

  Exercise difficulty progression: Easy -> Medium -> Challenging

  ### 5. Self-Test Questions (5-10 questions)
  Mix of:
  - Conceptual questions (test understanding)
  - True/False with explanations
  - Fill-in-the-blank for formulas
  - Short answer questions
  - Code reading questions (what does this output?)

  ### 6. Common Mistakes & Pitfalls
  - List common errors students make
  - How to avoid or debug them

  ### 7. Further Reading
  - Links to documentation
  - Related papers or resources
  - Advanced topics to explore

  **Style Guidelines**:
  - Use Typst syntax (similar to existing chapter files)
  - Include working Julia code blocks with `julia` language tag
  - Keep explanations concise but complete
  - Use consistent terminology with the source chapter
  - Target 30-45 minute completion time

  **Output**: Create file `learning-sheets/{chapter_name}-learning-sheet.typ`
```

## Step 2: Spawn Reviewer Agent

After Writer completes, spawn the Reviewer agent:

```
Prompt: |
  You are a learning materials reviewer for SCFP.

  **Task**: Review the learning sheet for {chapter_name}.

  **Inputs**:
  1. `learning-sheets/{chapter_name}-learning-sheet.typ` - The file to review
  2. `syllabus/syllabus.typ` - Verify learning objectives match the syllabus
  3. Source chapter in `book/chap*/` - Verify content accuracy
  4. Other learning sheets - Compare quality and style

  **Review Checklist**:

  ### Syllabus Alignment
  - [ ] Learning objectives match those in syllabus/syllabus.typ
  - [ ] Prerequisites are correctly identified per syllabus
  - [ ] Content scope matches syllabus week description

  ### Content Accuracy
  - [ ] All concepts match the source chapter
  - [ ] Mathematical formulas are correct
  - [ ] Code examples are syntactically correct
  - [ ] No factual errors or misconceptions

  ### Pedagogical Quality
  - [ ] Learning objectives are clear and measurable
  - [ ] Concepts build logically from simple to complex
  - [ ] Exercises have appropriate difficulty progression
  - [ ] Self-test questions cover all key concepts
  - [ ] Common mistakes section is helpful

  ### Code Quality
  - [ ] All code blocks specify language (```julia)
  - [ ] Code follows Julia best practices
  - [ ] Solutions are efficient and idiomatic
  - [ ] Comments explain non-obvious parts

  ### Completeness
  - [ ] All 7 sections are present
  - [ ] Prerequisites are correctly identified
  - [ ] Further reading links are relevant

  ### Style
  - [ ] Consistent with SCFP book style
  - [ ] Typst syntax is correct
  - [ ] No broken links or references

  **Output one of**:
  - `APPROVED` - All checks pass, ready for verification
  - `REVISE: <list>` - Specific issues with line references
  - `ESCALATE: <blocker>` - Issue requiring human decision
```

## Step 3: Spawn Verification Agent

After Reviewer approves, spawn the Verification agent:

```
Prompt: |
  You are a verification agent for SCFP learning materials.

  **Task**: Verify the learning sheet for {chapter_name}.

  ## 1. CODE VERIFICATION (CRITICAL)

  For EVERY code block:
  - Check syntax correctness
  - Verify imports and dependencies exist
  - Check that expected output matches actual behavior
  - Test edge cases mentioned in exercises

  ## 2. FACT-CHECK

  For factual claims:
  - Verify algorithm complexities are correct
  - Check mathematical statements
  - Verify package names and function signatures
  - Confirm external links work

  ## 3. CONSISTENCY CHECK

  - Terminology matches source chapter
  - Notation is consistent throughout
  - Difficulty levels are appropriate
  - Time estimate is realistic

  ## 4. COMPLETENESS CHECK

  - All learning objectives are addressed by content
  - All self-test questions have clear answers
  - All exercises have working solutions

  **Output one of**:
  - `VERIFIED` - All checks pass
  - `ISSUES: <list>` - List each issue with file:line, type, description, fix
  - `ESCALATE: <blocker>` - Unresolvable issue
```

## Step 4: Handle Review/Verification Output

- **APPROVED + VERIFIED**: Proceed to Step 5
- **REVISE/ISSUES**: Send feedback to Writer, re-run pipeline (max 3 cycles)
- **ESCALATE**: Stop and present issue to human

## Step 5: Compile and Verify

After APPROVED + VERIFIED:

```bash
# Create learning-sheets directory if needed
mkdir -p learning-sheets

# Compile the learning sheet
typst compile learning-sheets/{chapter_name}-learning-sheet.typ
```

If compilation fails, fix issues before committing.

## Step 6: Git Sync

After successful compilation:

```bash
git add learning-sheets/
git commit -m "Add learning sheet for {chapter_name}

Generated via subagent workflow:
- Learning objectives and prerequisites
- Key concepts summary
- Code exercises with solutions
- Self-test questions
- Common mistakes and further reading

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"

git push
```

## Escalation Triggers

Escalate to human when:
- 3 revision cycles without approval
- Source chapter content is unclear or incomplete
- Code examples require packages not in the project
- User hasn't confirmed scope/focus areas
- Significant discrepancies between source and standard practices

## Usage Examples

```
generate learning sheet for julia-basic
generate learning sheet for tensor-network
generate learning sheet for ad
```

This generates a comprehensive learning sheet for the specified chapter, validates it, and pushes to git.
