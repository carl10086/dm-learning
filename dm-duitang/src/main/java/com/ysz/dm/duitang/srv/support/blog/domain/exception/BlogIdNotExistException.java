package com.ysz.dm.duitang.srv.support.blog.domain.exception;

import com.ysz.dm.duitang.srv.infra.core.BizException;

public class BlogIdNotExistException extends BizException {

  public BlogIdNotExistException(final Long blogId) {
    super("blogId not exist:" + blogId);
  }
}
