# Workstream C — Data Engine

## Why this workstream exists

The schema and assumptions from Workstream B are just paper. The data engine actually produces the simulated dataset — loan records, customer attributes, repayment histories, channel attribution, behavioural signals — that every analytical workstream consumes. Without this, there is nothing to analyse.

## What this workstream does

Implements the data generation rules in code. Configures the rulebook that controls default calibration, segment behaviour, channel mix, macro conditions, and behavioural realism. Runs the generator deterministically (so the same inputs always produce the same outputs), validates that the generated data reconciles to the design intent, and freezes the data files for downstream use.

## What it produces

The loan-level, customer-level, repayment, and behavioural data files used by every downstream workstream.

## How it fits with other workstreams

Foundational. Workstreams D, E, F, G, and H all read from the files this workstream produces. If anything in the data is wrong, every later workstream inherits the error — which is why this workstream invests heavily in validation before signing off and freezing the outputs.
