package com.ysz.dm.jraft.counter;

import com.alipay.sofa.jraft.Node;
import com.alipay.sofa.jraft.RaftGroupService;

public class CounterServer {


    private RaftGroupService raftGroupService;
    private Node node;
    private CounterStateMachine fsm;

}
