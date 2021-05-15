package com.ysz.dm.fast.kernel.epoll;

import lombok.Getter;
import lombok.ToString;
import org.jetbrains.annotations.NotNull;

@Getter
@ToString
public class LinuxEpollRbItem {

    /**
     * 每个 item 的唯一 id
     */
    private final Long id;

    /**
     * 关联的 epoll 实例
     */
    private final LinuxEpoll epoll;

    public LinuxEpollRbItem(Long id, LinuxEpoll epoll) {
        this.id = id;
        this.epoll = epoll;
    }

    /**
     * 数据发生的时候
     *
     * @param data
     */
    public void onData(String data) {

    }


}
