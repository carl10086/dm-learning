package com.ysz.dm.duitang.srv.support.blog.domain.repo;

import com.ysz.dm.duitang.srv.support.blog.domain.model.Blog;

public interface BlogRepo {

  /**
   * 发布的时候、增加一整个 blog, 包含 原发和转发信息
   */
  void save(Blog blog);

  Blog findOne(Long forwardId);

  /**
   *
   * 转发的时候增加一个转发
   */
  void addForward(Blog blog);

  void changeAuditStatus(Blog blog);
}
