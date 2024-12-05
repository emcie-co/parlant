#!/bin/bash
report_num=$1
mkdir -p disk_reports

echo "=== Directory Sizes (1M+) ===" > "disk_reports/report_${report_num}.txt"
sudo du -hx --threshold=1M --exclude=/proc --exclude=/sys / 2>/dev/null | sort -hr >> "disk_reports/report_${report_num}.txt"

echo -e "\n=== Filesystem Information ===" >> "disk_reports/report_${report_num}.txt"
df -h >> "disk_reports/report_${report_num}.txt"

echo -e "\n=== Large Files (1M+) ===" >> "disk_reports/report_${report_num}.txt"
sudo find / -type f -size +1M -exec ls -lh {} \; 2>/dev/null | sort -k5 -hr >> "disk_reports/report_${report_num}.txt"

echo -e "\n=== Top Space Usage Directories ===" >> "disk_reports/report_${report_num}.txt"
sudo du /usr/ -hx -d 4 --threshold=1G 2>/dev/null | sort -hr | head -20 >> "disk_reports/report_${report_num}.txt"

echo -e "\n=== Top Packages by Size ===" >> "disk_reports/report_${report_num}.txt"
dpkg-query -Wf '${Installed-Size}\t${Package}\n' | sort -nr | head -20 >> "disk_reports/report_${report_num}.txt"
