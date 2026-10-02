use csv::Reader;
use std::error::Error;

/*
Gaurav Sablok
gsablok@proton.me
*/

#[tokio::main]
pub async fn csvprepapre(pathfile: &str) -> Result<Vec<Vec<f32>>, Box<dyn Error>> {
    let mut rdr = Reader::from_path(pathfile).unwrap();

    let mut columns: Vec<Vec<f32>> = Vec::new();

    for result in rdr.records() {
        let record = result?;
        if columns.is_empty() {
            columns = vec![Vec::new(); record.len()];
        }

        for (i, field) in record.iter().enumerate() {
            columns[i].push(field.to_string().parse::<f32>().unwrap());
        }
    }

    let mut final_columns: Vec<Vec<f32>> = Vec::new();

    for i in columns.iter() {
        for val in i.iter() {
            let mut vechold: Vec<f32> = Vec::new();
            if *val == 0f32 {
                continue;
            } else if *val != 0f32 {
                vechold.push(val.clone())
            }
            final_columns.push(vechold);
        }
    }
    Ok(final_columns)
}
