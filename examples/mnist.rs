use mnist::*;

fn main() {
    println!("mnist demo");

    // Deconstruct the returned Mnist struct.
    let Mnist {
        trn_img,
        trn_lbl,
        tst_img,
        tst_lbl,
        ..
    } = MnistBuilder::new()
        .download_and_extract() // this one is broken
        .label_format_digit()
        .training_set_length(50_000)
        // .validation_set_length(10_000)
        .test_set_length(10_000)
        .finalize();

    println!("{trn_img:?}")
}
