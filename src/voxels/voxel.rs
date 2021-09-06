
pub enum Side {
    Left, Right, Top, Bottom, Front, Back
}

pub type Chunk = Vec<[[Block; 16]; 16]>;

pub type World = Vec<Chunk>;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Block {
    AIR, GRASS, DIRT, STONE, LOG
}
