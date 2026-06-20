# Workstream E — Customer Segmentation


## Why this workstream exists

The engagement charter asks which customers are most likely to default and which need different policy treatment. A book of tens of thousands of loans is too coarse to act on as one bucket — policy has to land at the segment level. Without segmentation, the recommendation can only be one-size-fits-all, which is rarely the right answer for a portfolio that spans several different lending products and risk profiles.

## What this workstream does

Designs a segmentation that splits the book into a small number of segments that behave coherently — similar default rates, similar economics, similar customer profiles. Tests the segmentation rigorously: does it actually separate risk and value, or is it just labelling slices that already existed? Assigns a policy posture to each segment — whether the right action is to grow it, maintain it, contain it, or exit it.

## What it produces

A segment-assignment file (one segment per loan), segment-level metrics, and a recommended policy posture per segment.

## How it fits with other workstreams

Foundational for the analytical workstreams downstream. Workstream F (EWS) uses these segments to overlay flag rates and watchlists. Workstream G (scenarios) uses them to scope each candidate lever. Workstream I (recommendations) uses the policy postures as the starting point for the recommendation narrative. If segmentation changes, every workstream that depends on it has to revisit.
