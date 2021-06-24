package com.ysz.biz;

public class Tmp {

  public static void main(String[] args) {
//    tstSql();
    tstSql();
  }

  private static void tstSql() {

    String sql = "";
    for (int i = 0; i < 16; i++) {
      sql += "drop table blog_origin_" + i + ";\n";
    }

    System.err.println(sql);

    String createTableSql = "";

    for (int i = 0; i < 16; i++) {
      createTableSql += "create table blog_origin_" + i + "\n"
          + "(\n"
          + "   id bigint not null,\n"
          + "   contentType tinyint not null,\n"
          + "   senderId bigint not null,\n"
          + "   albumId bigint not null,\n"
          + "   createAt timestamp not null DEFAULT CURRENT_TIMESTAMP ,\n"
          + "   photoId bigint not null,\n"
          + "   officalTag varchar(255) null,\n"
          + "   msgAsShort varchar(500) null,\n"
          + "   ocr varchar(255) null,\n"
          + "   ocrTextLen int   null,\n"
          + "   status int not null,\n"
          + "   updateAt timestamp not null DEFAULT CURRENT_TIMESTAMP ,\n"
          + "   metaVersion int default 0 not null,\n"
          + "   sourceLink varchar(255) null,\n"
          + "   sourceTitle varchar(255) null,\n"
          + "   sourceType varchar(255) null,\n"
          + "   syncOffset timestamp(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6),\n"
          + "   constraint blog_origin_0_pk primary key (id)\n"
          + ") ENGINE = InnoDB DEFAULT CHARSET = utf8mb4; \n";
    }

//    System.out.println(createTableSql);

    createTableSql = "";

    for (int i = 0; i < 16; i++) {
      createTableSql += "create table blog_origin_msg_" + i + "\n"
          + "(\n"
          + "    id  bigint   not null,\n"
          + "    msg longtext null,\n"
          + "    constraint blog_origin_msg_pk  primary key (id)\n"
          + ") ENGINE = InnoDB DEFAULT CHARSET = utf8mb4;\n";
    }

//    System.out.println(createTableSql);

    createTableSql = "";

    for (int i = 0; i < 16; i++) {
      createTableSql += "create table blog_origin_audit_detail_" + i + "\n"
          + "(\n"
          + "    partitionId        varchar(255)                        not null,\n"
          + "    blogId             bigint                              not null,\n"
          + "    contentType        tinyint                             not null,\n"
          + "    senderId           bigint                              not null,\n"
          + "    createAt           timestamp default CURRENT_TIMESTAMP not null,\n"
          + "    photoId            bigint                              not null,\n"
          + "    albumId            bigint                              not null,\n"
          + "    imgCheckSupplier   varchar(255)                        null,\n"
          + "    hitLabel           varchar(255)                        null,\n"
          + "    labelLevel         int                                 null,\n"
          + "    labelRate          varchar(255)                        null,\n"
          + "    hitWords           varchar(255)                        null,\n"
          + "    machineStatus      tinyint                             null,\n"
          + "    machineAt          timestamp default CURRENT_TIMESTAMP not null,\n"
          + "    machineMsgStatus   tinyint                             null,\n"
          + "    machinePhotoStatus tinyint                             null,\n"
          + "    status             tinyint                             not null,\n"
          + "    firstAt            TIMESTAMP default CURRENT_TIMESTAMP not null,\n"
          + "    firstOp            bigint                              null,\n"
          + "    firstStatus        tinyint                             not null,\n"
          + "    secondAt           timestamp default CURRENT_TIMESTAMP not null,\n"
          + "    secondOp           bigint                              null,\n"
          + "    secondStatus       tinyint                             null,\n"
          + "    finalStatus        tinyint                             null,\n"
          + "    finalAuditType     tinyint                             null,\n"
          + "    finalOperator      bigint                              null,\n"
          + "    syncOffset         timestamp(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6),\n"
          + "    finalOperateAt     timestamp default CURRENT_TIMESTAMP not null,\n"
          + "    lastCheckNo        bigint                              null,\n"
          + "    lastCheckAt        timestamp                           null,\n"
          + "    lastCheckBy        bigint                              null,\n"
          + "    lastCheckAction    tinyint                             null,\n"
          + "    constraint blog_origin_audit_detail_pk\n"
          + "        primary key (blogId)\n"
          + ") ENGINE = InnoDB DEFAULT CHARSET = utf8mb4;\n"
          + "\n";
    }
//    System.out.println(createTableSql);
  }
}
