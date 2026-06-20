# Workstream B — Data Design

## Why this workstream exists

The engagement uses simulated data as specified in the deliverables. Before that data can be generated, somebody has to design the schema — what fields exist on each loan, customer, repayment, and channel record. The schema has to be rich enough that every question the charter promised can actually be answered from the data, and constrained enough that the data tells a coherent story instead of a random one.

## What this workstream does

Lays out the table structures, the relationships between them, and the field-level definitions for the simulated loan portfolio. Also locks the engagement's assumptions log — the single reference list of every assumption (cost of funds, loss given default, operating cost, macro stress band, and so on) that downstream workstreams will use. Without a single source of truth for assumptions, different workstreams would silently use different numbers and the final report would not reconcile.

## What it produces

A data design specification document and an assumptions log spreadsheet.

## How it fits with other workstreams

Hands off to Workstream C, which builds the actual data from this specification. Every downstream analytical workstream consumes data shaped according to this schema and refers back to the assumptions log when it needs a parameter value.
