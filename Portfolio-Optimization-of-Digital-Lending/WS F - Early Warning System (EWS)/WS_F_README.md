# Workstream F — Early Warning System

## Why this workstream exists

Workstreams D and E identify who is risky which channels lose money, which segments hold the bad apples. But that picture is static: it describes a loan's risk at origination, or across its whole life after the fact. It cannot tell you that a loan which looked fine three months ago is quietly deteriorating right now, while there is still time to act. By the time a loan reaches 90+ days past due the loss is essentially locked in; the decision has already been made for you. Workstream F adds the dimension the rest of the analysis is missing time. It watches performing loans month by month and flags the ones heading toward default early enough that something can still be done. It also rescues a pattern Workstream E found but could not use: a behavioural-distress cohort whose signals coincided with default rather than preceding it. F turns that pattern into a forward-looking flag that leads the event instead of confirming it.

## What this workstream does

Builds a dynamic, point-in-time score on every still-performing loan, refreshed monthly, from behavioural signals i.e cashflow consistency, balance volatility, utilisation creep, repayment-bounce streaks, spending shocks and each engineered so that it is observable strictly before the month it predicts, never coincident with it. The discipline that governs everything is no look-ahead: a feature may only see the past, and the model is validated out of time (trained on older months, tested on newer) so that the lead time it claims is real and not an artefact of hindsight. Loans that are already delinquent bypass the score and route straight to collections i.e the early-warning score is reserved for loans that still look healthy on the surface but are starting to behave like ones that won't. Each loan is graded green, amber or red, with the cut-offs set against how many cases the collections team can actually work and the cost of a missed default weighed against the cost of a false alarm.

## What it produces

An early-warning score definition; a current watchlist of flagged loans, each carrying its score, its red/amber/green band, its segment and a recommended action; a threshold table giving the expected monthly flag volume, precision and lead time at each band; and a validation pack that proves the flags fire before default, generalise out of time, and do not leak. Flag rates are reported by segment, with the behavioural-distress cohort surfaced at the top of the list.

## How it fits with other workstreams

Consumes Workstream E's segmentation, along with the behavioural and repayment data built by Workstream C to Workstream B's schema, drawing its cost parameters from the assumptions log. It operationalises the monitoring posture E recommended for the Contain segment turning "watch these loans" from a slogan into a ranked, actionable list. It complements Workstream G: where G's levers decline or reprice risk at the gate, F manages the risk already on the book, and the early-warning capability can itself be costed as a lever and pressure-tested for assumption sensitivity by Workstream H. Hands off to Workstream I, where the watchlist and its lead-time evidence become part of the recommendation packaged for the CRO.
