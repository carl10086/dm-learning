package com.ysz.dm.duitang.srv.support.blog.domain;

public class BlogDetail {

  private final BlogForward blogForward;

  private BlogCounter blogCounter;
  private BlogFile blogFile;

  public BlogDetail(final BlogForward blogForward) {
    this.blogForward = blogForward;
  }
}
