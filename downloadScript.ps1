$prefix =  'SOLAR_DATA'
$path = 'ROC/SOLAR_DATA'
$start_date = '2016-01-01'
$end_date = '2022-08-01'
$item_array = @('Tirrawarra', 'Mount_Margaret', 'Moobam Jackson', 'Epsilon', 'Durham', 'Daralingie')

foreach ($item in $item_array) {
    python downloadS3.py -item_cd $item -prefix $prefix -path $path -start_date $start_date -end_date $end_date -mode weather 
}