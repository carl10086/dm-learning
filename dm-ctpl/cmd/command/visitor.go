package command

import (
	"bufio"
	"github.com/spf13/cobra"
	"io"
	"k8s.io/apimachinery/pkg/util/yaml"
	"os"
)

type YAMLVisitor interface {
	Visit(func(yamlDoc []byte) error) error
	Close()
}

type yamlVisitor struct {
	reader io.Reader
}

// Visit implements YAMLVisitor
func (v *yamlVisitor) Visit(fn func(yamlDoc []byte) error) error {
	r := yaml.NewYAMLReader(bufio.NewReader(v.reader))

	for {
		data, err := r.Read()
		if len(data) == 0 {
			if err == io.EOF {
				return nil
			}
			if err != nil {
				return err
			}
			continue
		}
		if err = fn(data); err != nil {
			return err
		}
	}
}

// Close closes the yamlVisitor
func (v *yamlVisitor) Close() {
	if closer, ok := v.reader.(io.Closer); ok {
		closer.Close()
	}
}

func buildYAMLVisitor(yamlFile string, cmd *cobra.Command) YAMLVisitor {
	var r io.ReadCloser
	if yamlFile == "" {
		r = io.NopCloser(os.Stdin)
	} else if f, err := os.Open(yamlFile); err != nil {
		ExitWithErrorf("%s failed: %v", cmd.Short, err)
	} else {
		r = f
	}
	return &yamlVisitor{reader: r}
}
