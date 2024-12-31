# Overview
`space_graph` is a package for visualizing 3D stuff

## Requirements
`pip install pygame moderngl`

# Developer Notes

When working with `Uniforms`, it is often helpful to see what `Uniforms` have been mapped - this can be done with
```python
for uniform_name, uniform_type in shader_program._members.items():
    print(uniform_name, uniform_type)
```
