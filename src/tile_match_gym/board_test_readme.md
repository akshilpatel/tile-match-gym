# Readme for the testing json file

Each of the test entries has the following elements:
- "name"
    - describes the starting board
    - "vertical 4" means that there is a vertical 4-in-a-row to be matched in
      the board
- "board"
    - the 2D description of the current board state
- "matches"
    - the 3-in-a-row matches that should possibly be activated on the board
- "islands"
    - used to distinguish L shaped matches where a larger contiguous group is
      identified
- "tile_names"
    - the type of special that will be produced after processing the board
- "tile_locations"
    - where a given special tile is being produced from
    - for example, if generating a cookie from a 5-in-a-row, this will contain 5
      coordinate values
- "first_activation"
    - The expected board state after the first activation (before any specials are
      activated)
- "activation_q"
    - The expected special tiles that are due to be activated
- "post_activation"
    - the expect after effect of activating the head of the activation queue
- "post_gravity"
    - the expected after effect of applying the gravity function
- "post_refill"
    - the expected board after refilling empty positions
