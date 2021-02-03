package com.ysz.dm.duitang.srv.support.blog.application.impl;

import com.ysz.dm.duitang.cli.support.blog.dto.DetailQueryCond;
import com.ysz.dm.duitang.srv.support.blog.domain.adapter.BlogCountAdapter;
import com.ysz.dm.duitang.srv.support.blog.domain.adapter.BlogFileAdapter;
import com.ysz.dm.duitang.srv.support.blog.domain.adapter.BlogUserAdapter;
import com.ysz.dm.duitang.srv.support.blog.domain.model.Blog;
import com.ysz.dm.duitang.srv.support.blog.domain.repo.BlogRepo;
import com.ysz.dm.duitang.srv.support.blog.domain.presentation.BlogDetailPresentation;

public class BlogAdmReadAppSrvImpl {

  private BlogRepo blogRepo;

  private BlogCountAdapter blogCountAdapter;

  private BlogFileAdapter blogFileAdapter;

  private BlogUserAdapter blogUserAdapter;

  private BlogDetailPresentation findBlog(
      DetailQueryCond query,
      Long blogId
  ) {
    final Blog blog = blogRepo.findOne(blogId);
    if (blog == null) {
      return null;
    }

    BlogDetailPresentation res = new BlogDetailPresentation(blog);
    if (query.isFillCount()) {
      // fill Count info
    }

    if (query.isFillFile()) {
      // fill file info
    }

    if (query.isFillUser()) {
      // fill user info
    }

    return res;
  }

}
