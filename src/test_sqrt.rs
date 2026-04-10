#![no_std]
pub fn test(a: f32) -> f32 {
    #[cfg(feature = "std")]
    { a.sqrt() }
    #[cfg(not(feature = "std"))]
    { a } // fallback for now
}
