package com.ysz.dm.duitang.srv.support.blog.domain.model;

import java.util.List;

public class BlogText {

  private final String msg;

  private List<String> tags;

  public BlogText(final String msg) {
    chkMsg();
    this.msg = msg;
  }

  private void chkMsg() {
    if (msg == null) {
      // ?
    }
  }
}
