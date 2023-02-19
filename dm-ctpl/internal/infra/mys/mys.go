package mys

import (
	"database/sql"
	"fmt"
	_ "github.com/go-sql-driver/mysql"
	"time"
)

type MysqlConfig struct {
	Addr     string
	Username string
	Password string
}

type ColumnMeta struct {
	ColumnName       string
	DataType         string
	ColumnType       string
	ColumnDefault    sql.NullString
	Nullable         string
	ColumnComment    string
	NumericPrecision sql.NullInt64
}

func FetchTableMetas(
	config *MysqlConfig,
	database string,
	table string,
) ([]*ColumnMeta, error) {
	//db, err := sql.Open("mysql", "adm:oK1@cM2]dB2!@tcp(10.200.68.3:3306)/zcwdb?tls=skip-verify&charset=utf8mb4,utf8")
	db, err := sql.Open(
		"mysql",
		fmt.Sprintf("%s:%s@tcp(%s)/?tls=skip-verify&charset=utf8mb4,utf8",
			config.Username,
			config.Password,
			config.Addr))
	if err != nil {
		return nil, err
	}
	// See "Important settings" section.
	db.SetConnMaxLifetime(time.Minute * 3)
	db.SetMaxOpenConns(1)
	db.SetMaxIdleConns(1)

	defer db.Close()
	stmtOut, err := db.Prepare(fmt.Sprintf(
		"SELECT COLUMN_NAME, DATA_TYPE,  COLUMN_TYPE, COLUMN_DEFAULT, IS_NULLABLE, COLUMN_COMMENT, NUMERIC_PRECISION FROM information_schema.columns WHERE table_schema=? and table_name = ?",
	))

	if err != nil {
		return nil, err
	}

	defer stmtOut.Close()
	rs, err := stmtOut.Query(database, table)
	if err != nil {
		return nil, err
	}

	metas := make([]*ColumnMeta, 0, 16)
	for rs.Next() {
		var meta ColumnMeta

		rs.Scan(&meta.ColumnName, &meta.DataType, &meta.ColumnType, &meta.ColumnDefault, &meta.Nullable, &meta.ColumnComment, &meta.NumericPrecision)
		metas = append(metas, &meta)
	}

	return metas, nil
}
