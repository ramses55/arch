from utils_disk import push_to_queue, ls_db, mkdir


#makes sure certain dirs do exist
mkdir('ok')
mkdir('failed')
mkdir('ok/orig')
mkdir('ok/marked')
mkdir('failed/marked')
mkdir('failed/orig')


files = ls_db("")
push_to_queue(files)
print(files)
if files:
    push_to_queue([['-','-']])
