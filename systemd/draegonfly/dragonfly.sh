#/bin/bash
#
# a bash script for dragonfly .


SYS_HOME="/duitang/dist/sys/dragonfly"
PID_FILE="${SYS_HOME}/run.pid"
EXEC_CMD="$SYS_HOME/dragonfly-x86_64 --cache_mode=true --port=6379 --bind=10.200.16.20 --dir=/data0 --maxmemory=0"

start() {
  if [ -f "${PID_FILE}" ]; then
    echo "pid file already exist"
    exit 1
  fi
  nohup ${EXEC_CMD} > ${SYS_HOME}/stdout.log 2>&1 &
#  disown $!
  echo $! > ${PID_FILE}
  echo "start success"
}


stop() {
  if [ ! -f "${PID_FILE}" ]; then
    echo "pid file not existed"
    exit 1
  fi
  PID=`cat ${PID_FILE}`
  echo "pid is $PID"
  kill $PID
  rm ${PID_FILE}
  echo "STOP SUCCESS"
}

usage () {
  echo "Usage"
  echo "$0 [-h] [start/stop/restart/reload]"
}


for i  in `seq 1 $#`
do
 case $1 in
        start)
            start
            exit 0
            ;;
        stop)
            stop
            exit 0
            ;;
        restart)
            stop
            start
            exit 0
            ;;
        reload)
            reload;
            exit 0
            ;;
         *)
          usage
          ;;
    esac
done