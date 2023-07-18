# Game Rules

This documents the game mechanics we have in place for the game. This is a living document and will be updated as we add more features to the game.

## Game setup

We can specify a board a height, width and number of tile colours, and types of special tiles to include in the game.
A board is generated with random (ordinary) tile colours such that no existing matches are present.

## Overall Game loop

## Detecting Matching

Matches are detected and resolved from the bottom up. The board is scanned row by row from the bottom until it finds a row with a match. Then only matches containing a tile in the lowest row are consider for match resolution. We do this because on resolving matches at the lowest row, the board cascades and refills which may change the matches in the rows above.

The code itself finds lines of length 3 or more. Then the matches are extended and merged where they may intersect to form more complex _islands_. Following this, matches are extracted (removing tiles fro the detected island) according to a priority order over match types. The priority order depends on the number of special tiles in the game.

## Resolving Matches

Each match is resolved in a creation, activation, gravity, refill loop.

Given a match, we first detect whether the match should create a new special tile. If so, the new tile is remembered to be created after the activation loop is processed.

If there is a special tile to be created, we remember this for after activation. Each tile in the match is marked for activation by appending it to an activation queue.

The tiles in the match are activated (as well as subsequent activations caused by chaining) and then the new tile is placed in the board.

### Special Candy Creation

Candy matches are created according to a priority list.

## Gravity

Gravity works currently as a downwards force.

## Refilling

All empty slots are filled with randomly coloured ordinary tiles.
