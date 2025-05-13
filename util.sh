for job in $(qstat -u iferreira | awk 'NR>5 {print $1}'); do
    echo -n "$job: "
    qstat -f "$job" 2>/dev/null | grep 'Job_Name =' | awk -F'= ' '{print $2}'
done
