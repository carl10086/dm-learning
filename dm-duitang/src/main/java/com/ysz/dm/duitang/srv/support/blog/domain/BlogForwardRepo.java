package com.ysz.dm.duitang.srv.support.blog.domain;

public interface BlogForwardRepo {

  void save(BlogForward blogForward);

  BlogForward findOne(Long forwardId);
}
