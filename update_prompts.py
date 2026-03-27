#!/usr/bin/env python3
"""Script to update the prompts.py file with updated rules 10 and 11."""

with open('prompts.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace the rules
old_rules = """10. **Step quality (strict)**: Instructions must be concrete and detailed, not vague. \\
      Each step should include clear action, timing/heat cues, and expected result.
11. **Beverage pairing**: Always include a `beverage_pairing` object, even for beverage-only requests \\
    (in that case, suggest a complementary snack instead and note it in the recipe)."""

new_rules = """10. **Step quality (strict)**: ALL instruction fields MUST be arrays of strings (lists). This includes \\
      both `step_by_step_instructions` and `beverage_pairing.instructions`. Each step should be \\
      concrete, detailed, and not vague. Each array element should include clear action, timing/heat cues, and expected result.
11. **Beverage pairing**: Always include a `beverage_pairing` object with an `instructions` field that \\
    MUST be an array of strings (each string is one preparation step), even for beverage-only requests \\
    (in that case, suggest a complementary snack instead and note it in the recipe)."""

if old_rules in content:
    content = content.replace(old_rules, new_rules)
    with open('prompts.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print("✓ Successfully updated prompts.py")
else:
    print("✗ Could not find the exact text to replace")
    print("Looking for:")
    print(repr(old_rules))
