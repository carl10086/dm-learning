package com.ysz.dm.duitang.srv.support.blog.domain;

import com.ysz.dm.duitang.cli.support.blog.dto.ForwardBlogReq;
import com.ysz.dm.duitang.srv.infra.core.BizException;

public class BlogForwardNotForbiddenException extends BizException {


    private final ForwardBlogReq req;

    public BlogForwardNotForbiddenException(final ForwardBlogReq req) {
        this.req = req;
    }
}
