package com.ysz.dm.duitang.srv.support.blog.domain.presentation;

import com.ysz.dm.duitang.srv.support.blog.domain.adapter.BlogCount;
import com.ysz.dm.duitang.srv.support.blog.domain.adapter.BlogFile;
import com.ysz.dm.duitang.srv.support.blog.domain.adapter.BlogUser;
import com.ysz.dm.duitang.srv.support.blog.domain.model.Blog;
import lombok.Setter;

public class BlogDetailPresentation {

  private final Blog blog;

  @Setter
  private BlogCount blogCount;
  @Setter
  private BlogFile blogFile;
  @Setter
  private BlogUser blogUser;

  public BlogDetailPresentation(final Blog blog) {
    this.blog = blog;
  }
}
