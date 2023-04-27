diag:
	python main.py type=create_diagnosis

all:
	python main.py -m \
		type=metric_curve,metric_value,metric_value,overall_ranks,performance_profiles,probability_of_improvement,sample_efficiency_curve

curve:
	python main.py type=metric_curve

value:
	python main.py type=metric_value

rank:
	python main.py type=overall_ranks

profile:
	python main.py type=performance_profiles

imporve:
	python main.py type=probability_of_improvement

efficiency:
	python main.py type=sample_efficiency_curve

## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Lint using flake8
lint:
	flake8 src
