# Per-map self-improvement, GPT-6 Luna teacher

Each cell: teacher calls / tree share / decision age (cycles) / ending right of the repeats / steps, mean over the layout's complete repeats. F: the tree after episode 6, frozen, no model.

| layout | situations | repeats | ep 1 | ep 2 | ep 3 | ep 4 | ep 5 | ep 6 | F |
|---|---|---|---|---|---|---|---|---|---|
| 300 | wall | 3 | 11.0 / 0% / 7.9 / 3/3 / 144 | 6.3 / 54% / 4.1 / 3/3 / 166 | 7.7 / 70% / 3.0 / 3/3 / 212 | 1.7 / 89% / 1.7 / 3/3 / 168 | 0.3 / 97% / 1.2 / 3/3 / 149 | 0.3 / 97% / 1.2 / 3/3 / 149 | 67% / 2/3 / 271; -; REPLAN_ROUTE,REORIENT; REPLAN_ROUTE,REORIENT |
| 302 | wall + gate | 3 | 31.3 / 0% / 7.3 / 3/3 / 357 | 7.0 / 78% / 2.6 / 3/3 / 356 | 2.3 / 93% / 1.5 / 3/3 / 355 | 0.0 / 100% / 1.0 / 2/3 / 315 | 1.3 / 96% / 1.4 / 3/3 / 362 | 1.0 / 97% / 1.2 / 3/3 / 362 | 100% / 3/3 / 362; REPLAN_ROUTE,WAIT,WAIT,WAIT; REPLAN_ROUTE,WAIT,WAIT,WAIT,WAIT; REPLAN_ROUTE,WAIT,WAIT,WAIT,WAIT |
| 304 | gate + gate | 3 | 42.3 / 0% / 7.5 / 2/3 / 480 | 9.0 / 78% / 2.5 / 1/3 / 416 | 3.3 / 91% / 1.4 / 1/3 / 416 | 4.3 / 89% / 1.6 / 1/3 / 416 | 5.3 / 87% / 1.7 / 1/3 / 416 | 2.7 / 92% / 1.4 / 1/3 / 416 | 71% / 1/3 / 416; WAIT,WAIT,RESUME_ROUTE; RESUME_ROUTE,WAIT,WAIT,WAIT,REORIENT,RESUME_ROUTE,REPLAN_ROUTE; RESUME_ROUTE,REORIENT,WAIT,WAIT,WAIT,REORIENT,REPLAN_ROUTE,RESUME_ROUTE,RESUME_ROUTE,REORIENT,WAIT,WAIT,REORIENT |
| 305 | reversed + wall + gate | 3 | 23.3 / 0% / 7.4 / 2/3 / 278 | 9.3 / 72% / 2.6 / 2/3 / 336 | 3.3 / 92% / 1.4 / 2/3 / 339 | 0.3 / 99% / 1.0 / 2/3 / 339 | 1.3 / 93% / 1.3 / 2/3 / 300 | 4.3 / 88% / 1.7 / 2/3 / 315 | 90% / 2/3 / 339; REPLAN_ROUTE,WAIT,WAIT,WAIT,WAIT; REPLAN_ROUTE,RESUME_ROUTE,REORIENT,REORIENT,REORIENT,REPLAN_ROUTE,RESUME_ROUTE,WAIT,WAIT,WAIT; REPLAN_ROUTE,WAIT,WAIT,WAIT,WAIT |
| 307 | wall + gate + gate | 3 | 44.7 / 0% / 6.7 / 1/3 / 478 | 18.0 / 58% / 4.1 / 0/3 / 500 | 4.3 / 90% / 1.6 / 0/3 / 500 | 3.3 / 93% / 1.4 / 1/3 / 478 | 2.7 / 94% / 1.4 / 0/3 / 500 | 2.0 / 96% / 1.2 / 0/3 / 500 | 70% / 1/3 / 392; REPLAN_ROUTE; REPLAN_ROUTE,WAIT,RESUME_ROUTE,RESUME_ROUTE,RESUME_ROUTE,RESUME_ROUTE,RESUME_ROUTE,REORIENT,WAIT,WAIT; REPLAN_ROUTE,WAIT,WAIT,WAIT,WAIT,WAIT,WAIT |
| 315 | breakdown (human) | 3 | 15.0 / 0% / 7.3 / 3/3 / 179 | 5.0 / 64% / 3.2 / 3/3 / 165 | 1.7 / 88% / 2.1 / 3/3 / 176 | 0.7 / 95% / 1.5 / 3/3 / 168 | 0.3 / 98% / 1.2 / 3/3 / 174 | 0.0 / 100% / 1.0 / 3/3 / 170 | 100% / 3/3 / 170; WAIT,WAIT,WAIT,WAIT,WAIT,REQUEST_HUMAN; WAIT,WAIT,WAIT,WAIT,WAIT,REQUEST_HUMAN; WAIT,REQUEST_HUMAN |
| 319 | reversed + wall | 3 | 15.3 / 0% / 7.6 / 3/3 / 194 | 3.7 / 76% / 2.5 / 3/3 / 174 | 0.7 / 96% / 1.2 / 3/3 / 171 | 0.0 / 100% / 1.0 / 3/3 / 170 | 0.0 / 100% / 1.0 / 3/3 / 170 | 0.0 / 100% / 1.0 / 3/3 / 170 | 100% / 3/3 / 170; REPLAN_ROUTE; REPLAN_ROUTE; REPLAN_ROUTE |
| 461 | sealed (human) | 3 | 2.0 / 0% / 9.0 / 3/3 / 45 | 2.0 / 0% / 8.0 / 3/3 / 44 | 0.0 / 100% / 1.0 / 3/3 / 37 | 0.0 / 100% / 1.0 / 3/3 / 37 | 0.0 / 100% / 1.0 / 3/3 / 37 | 0.0 / 100% / 1.0 / 3/3 / 37 | 100% / 3/3 / 37; REPLAN_ROUTE,REQUEST_HUMAN; REPLAN_ROUTE,REQUEST_HUMAN; REPLAN_ROUTE,REQUEST_HUMAN |

Over layouts (mean of the layout means; runs ending right out of all runs):

| episode | calls | tree share | age | right | steps |
|---|---|---|---|---|---|
| 1 | 23.12 | 0% | 7.60 | 20/24 | 269 |
| 2 | 7.54 | 60% | 3.71 | 18/24 | 269 |
| 3 | 2.92 | 90% | 1.66 | 18/24 | 276 |
| 4 | 1.29 | 96% | 1.28 | 18/24 | 261 |
| 5 | 1.42 | 96% | 1.29 | 18/24 | 263 |
| 6 | 1.29 | 96% | 1.22 | 18/24 | 265 |
| F | 0 | 87% | 1.00 | 18/24 | 270 |

Teacher calls in all: 902 over 24 runs.
